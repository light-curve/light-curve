use conv::ConvUtil;
use include_dir::{include_dir, Dir};
use light_curve_feature::{Float, TimeSeries};
use serde::Deserialize;
use unzip3::Unzip3;

#[derive(Deserialize)]
struct LightCurveRecord {
    ant_mjd: f64,
    ant_mag: f64,
    ant_magerr: f64,
    ant_passband: char,
    ant_survey: u8,
}

pub fn iter_sn1a_flux_ts<T>() -> impl Iterator<Item = (&'static str, TimeSeries<'static, T>)>
where
    T: Float,
{
    // Relative to the current file
    const ZTF_IDS_CSV: &str =
        include_str!("../../../test-data/SNIa/snIa_bandg_minobs10_beforepeak3_afterpeak4.csv");

    // Relative to the project root
    const LC_DIR: Dir = include_dir!("../../test-data/SNIa/light-curves");

    ZTF_IDS_CSV.split_terminator('\n').map(|ztf_id| {
        let filename = format!("{}.csv", ztf_id);
        let file = LC_DIR.get_file(&filename).unwrap();
        let mut reader = csv::ReaderBuilder::new().from_reader(file.contents());
        let (t, flux, w_flux): (Vec<_>, Vec<_>, Vec<_>) = reader
            .deserialize()
            .map(|row| row.unwrap())
            .filter(|record: &LightCurveRecord| {
                record.ant_passband == 'g' && record.ant_survey == 1
            })
            .map(|record| {
                let LightCurveRecord {
                    ant_mjd: t,
                    ant_mag: m,
                    ant_magerr: sigma_m,
                    ..
                } = record;
                let flux = 10.0_f64.powf(-0.4_f64 * m);
                let sigma_flux = f64::ln(10.0_f64) * 0.4_f64 * sigma_m * flux;
                let w_flux = sigma_flux.powi(-2);
                (
                    t.approx_as::<T>().unwrap(),
                    flux.approx_as::<T>().unwrap(),
                    w_flux.approx_as::<T>().unwrap(),
                )
            })
            .unzip3();
        (ztf_id, TimeSeries::new(t, flux, w_flux))
    })
}
