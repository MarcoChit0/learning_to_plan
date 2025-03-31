; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 964125

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child368 - child
    bread101 - bread-portion
    content68 - content-portion
    tray67 tray168 - tray
    table469 table214 table252 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray67 kitchen)
     (at tray168 kitchen)
     (at_kitchen_bread bread101)
     (at_kitchen_content content68)
     (not_allergic_gluten child368)
     (waiting child368 table252)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child368)
    )
  )
)
