; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 229590

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child242 child452 - child
    bread238 bread192 - bread-portion
    content165 content139 - content-portion
    tray475 tray263 tray264 tray367 - tray
    table206 table466 table196 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray475 kitchen)
     (at tray263 kitchen)
     (at tray264 kitchen)
     (at tray367 kitchen)
     (at_kitchen_bread bread238)
     (at_kitchen_bread bread192)
     (at_kitchen_content content165)
     (at_kitchen_content content139)
     (no_gluten_bread bread192)
     (no_gluten_content content139)
     (allergic_gluten child452)
     (not_allergic_gluten child242)
     (waiting child242 table206)
     (waiting child452 table196)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child242)
     (served child452)
    )
  )
)
