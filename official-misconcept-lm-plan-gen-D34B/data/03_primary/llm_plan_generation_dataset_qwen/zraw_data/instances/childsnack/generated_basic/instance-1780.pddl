; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 918010

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child316 - child
    bread155 - bread-portion
    content57 - content-portion
    tray22 - tray
    table299 table128 table365 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray22 kitchen)
     (at_kitchen_bread bread155)
     (at_kitchen_content content57)
     (not_allergic_gluten child316)
     (waiting child316 table299)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child316)
    )
  )
)
