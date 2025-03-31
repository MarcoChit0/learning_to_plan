; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 777061

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child291 child265 - child
    bread436 bread488 - bread-portion
    content450 content248 - content-portion
    tray127 tray497 - tray
    table369 table433 table447 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray127 kitchen)
     (at tray497 kitchen)
     (at_kitchen_bread bread436)
     (at_kitchen_bread bread488)
     (at_kitchen_content content450)
     (at_kitchen_content content248)
     (not_allergic_gluten child291)
     (not_allergic_gluten child265)
     (waiting child291 table433)
     (waiting child265 table447)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child291)
     (served child265)
    )
  )
)
