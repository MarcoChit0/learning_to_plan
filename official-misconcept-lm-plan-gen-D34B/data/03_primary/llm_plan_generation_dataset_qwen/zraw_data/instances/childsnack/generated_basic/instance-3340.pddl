; child-snack task with 2 children and 0.71 gluten factor 
; constant factor of 1.7
; random seed: 345964

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child182 child103 - child
    bread62 bread298 - bread-portion
    content300 content306 - content-portion
    tray66 tray346 tray117 tray141 - tray
    table350 table285 table376 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray66 kitchen)
     (at tray346 kitchen)
     (at tray117 kitchen)
     (at tray141 kitchen)
     (at_kitchen_bread bread62)
     (at_kitchen_bread bread298)
     (at_kitchen_content content300)
     (at_kitchen_content content306)
     (no_gluten_bread bread62)
     (no_gluten_content content300)
     (allergic_gluten child182)
     (not_allergic_gluten child103)
     (waiting child182 table285)
     (waiting child103 table376)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child182)
     (served child103)
    )
  )
)
