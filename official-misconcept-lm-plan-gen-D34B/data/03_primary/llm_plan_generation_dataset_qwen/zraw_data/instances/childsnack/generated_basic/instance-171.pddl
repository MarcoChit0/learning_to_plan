; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 157628

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child195 child327 - child
    bread94 bread133 - bread-portion
    content475 content68 - content-portion
    tray11 tray228 tray154 tray283 - tray
    table424 table293 table168 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray11 kitchen)
     (at tray228 kitchen)
     (at tray154 kitchen)
     (at tray283 kitchen)
     (at_kitchen_bread bread94)
     (at_kitchen_bread bread133)
     (at_kitchen_content content475)
     (at_kitchen_content content68)
     (not_allergic_gluten child195)
     (not_allergic_gluten child327)
     (waiting child195 table424)
     (waiting child327 table424)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child195)
     (served child327)
    )
  )
)
