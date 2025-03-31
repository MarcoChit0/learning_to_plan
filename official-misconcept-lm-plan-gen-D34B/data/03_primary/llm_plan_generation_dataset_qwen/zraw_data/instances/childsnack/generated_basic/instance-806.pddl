; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 478337

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child227 - child
    bread109 - bread-portion
    content309 - content-portion
    tray493 tray303 - tray
    table437 table210 table226 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray493 kitchen)
     (at tray303 kitchen)
     (at_kitchen_bread bread109)
     (at_kitchen_content content309)
     (not_allergic_gluten child227)
     (waiting child227 table437)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child227)
    )
  )
)
