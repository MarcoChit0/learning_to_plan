; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 6141

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child494 child459 - child
    bread216 bread381 - bread-portion
    content468 content279 - content-portion
    tray440 tray204 - tray
    table7 table445 table60 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray440 kitchen)
     (at tray204 kitchen)
     (at_kitchen_bread bread216)
     (at_kitchen_bread bread381)
     (at_kitchen_content content468)
     (at_kitchen_content content279)
     (not_allergic_gluten child494)
     (not_allergic_gluten child459)
     (waiting child494 table7)
     (waiting child459 table7)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child494)
     (served child459)
    )
  )
)
