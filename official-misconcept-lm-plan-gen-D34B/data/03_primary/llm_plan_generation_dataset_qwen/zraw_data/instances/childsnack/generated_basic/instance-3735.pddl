; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 301338

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child100 - child
    bread72 - bread-portion
    content368 - content-portion
    tray336 tray12 tray85 - tray
    table466 table486 table483 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray336 kitchen)
     (at tray12 kitchen)
     (at tray85 kitchen)
     (at_kitchen_bread bread72)
     (at_kitchen_content content368)
     (not_allergic_gluten child100)
     (waiting child100 table483)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child100)
    )
  )
)
