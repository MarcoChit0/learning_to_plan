; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 144681

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child129 child353 - child
    bread11 bread459 - bread-portion
    content455 content244 - content-portion
    tray318 tray401 tray457 - tray
    table408 table319 table460 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray318 kitchen)
     (at tray401 kitchen)
     (at tray457 kitchen)
     (at_kitchen_bread bread11)
     (at_kitchen_bread bread459)
     (at_kitchen_content content455)
     (at_kitchen_content content244)
     (no_gluten_bread bread459)
     (no_gluten_content content244)
     (allergic_gluten child129)
     (not_allergic_gluten child353)
     (waiting child129 table408)
     (waiting child353 table319)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child129)
     (served child353)
    )
  )
)
