; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 512179

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child445 - child
    bread232 - bread-portion
    content114 - content-portion
    tray398 - tray
    table287 table28 table440 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray398 kitchen)
     (at_kitchen_bread bread232)
     (at_kitchen_content content114)
     (not_allergic_gluten child445)
     (waiting child445 table287)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child445)
    )
  )
)
