; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 809752

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child345 - child
    bread445 - bread-portion
    content191 - content-portion
    tray100 tray383 - tray
    table100 table413 table428 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray100 kitchen)
     (at tray383 kitchen)
     (at_kitchen_bread bread445)
     (at_kitchen_content content191)
     (not_allergic_gluten child345)
     (waiting child345 table413)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child345)
    )
  )
)
