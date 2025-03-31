; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 860725

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child463 - child
    bread33 - bread-portion
    content469 - content-portion
    tray203 tray317 tray282 - tray
    table386 table418 table194 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray203 kitchen)
     (at tray317 kitchen)
     (at tray282 kitchen)
     (at_kitchen_bread bread33)
     (at_kitchen_content content469)
     (not_allergic_gluten child463)
     (waiting child463 table386)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child463)
    )
  )
)
