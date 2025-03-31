; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 634972

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child286 - child
    bread270 - bread-portion
    content5 - content-portion
    tray422 tray419 - tray
    table81 table456 table386 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray422 kitchen)
     (at tray419 kitchen)
     (at_kitchen_bread bread270)
     (at_kitchen_content content5)
     (not_allergic_gluten child286)
     (waiting child286 table386)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child286)
    )
  )
)
