; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 721379

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child324 child137 - child
    bread300 bread32 - bread-portion
    content68 content417 - content-portion
    tray53 tray156 tray304 - tray
    table353 table269 table53 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray53 kitchen)
     (at tray156 kitchen)
     (at tray304 kitchen)
     (at_kitchen_bread bread300)
     (at_kitchen_bread bread32)
     (at_kitchen_content content68)
     (at_kitchen_content content417)
     (no_gluten_bread bread32)
     (no_gluten_content content417)
     (allergic_gluten child324)
     (not_allergic_gluten child137)
     (waiting child324 table53)
     (waiting child137 table53)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child324)
     (served child137)
    )
  )
)
