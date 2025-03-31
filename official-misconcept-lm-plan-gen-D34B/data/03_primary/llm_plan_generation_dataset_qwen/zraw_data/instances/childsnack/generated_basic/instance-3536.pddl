; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 483223

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child204 child70 - child
    bread405 bread453 - bread-portion
    content452 content117 - content-portion
    tray182 tray298 - tray
    table65 table208 table50 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray182 kitchen)
     (at tray298 kitchen)
     (at_kitchen_bread bread405)
     (at_kitchen_bread bread453)
     (at_kitchen_content content452)
     (at_kitchen_content content117)
     (not_allergic_gluten child204)
     (not_allergic_gluten child70)
     (waiting child204 table50)
     (waiting child70 table208)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child204)
     (served child70)
    )
  )
)
