; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 732165

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child256 - child
    bread213 - bread-portion
    content218 - content-portion
    tray150 - tray
    table19 table148 table219 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray150 kitchen)
     (at_kitchen_bread bread213)
     (at_kitchen_content content218)
     (not_allergic_gluten child256)
     (waiting child256 table148)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child256)
    )
  )
)
