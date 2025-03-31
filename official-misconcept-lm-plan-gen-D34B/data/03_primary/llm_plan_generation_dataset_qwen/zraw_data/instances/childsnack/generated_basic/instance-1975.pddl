; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 732260

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child184 child28 - child
    bread319 bread312 - bread-portion
    content164 content496 - content-portion
    tray293 - tray
    table326 table330 table215 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray293 kitchen)
     (at_kitchen_bread bread319)
     (at_kitchen_bread bread312)
     (at_kitchen_content content164)
     (at_kitchen_content content496)
     (no_gluten_bread bread319)
     (no_gluten_content content496)
     (allergic_gluten child28)
     (not_allergic_gluten child184)
     (waiting child184 table215)
     (waiting child28 table330)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child184)
     (served child28)
    )
  )
)
