; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 62726

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child333 - child
    bread322 - bread-portion
    content141 - content-portion
    tray338 - tray
    table256 table18 table198 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray338 kitchen)
     (at_kitchen_bread bread322)
     (at_kitchen_content content141)
     (not_allergic_gluten child333)
     (waiting child333 table198)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child333)
    )
  )
)
