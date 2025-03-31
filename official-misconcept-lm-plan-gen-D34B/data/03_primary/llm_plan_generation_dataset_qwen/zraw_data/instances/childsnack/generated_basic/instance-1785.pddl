; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 715760

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child391 - child
    bread425 - bread-portion
    content269 - content-portion
    tray192 - tray
    table186 table179 table188 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray192 kitchen)
     (at_kitchen_bread bread425)
     (at_kitchen_content content269)
     (not_allergic_gluten child391)
     (waiting child391 table188)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child391)
    )
  )
)
