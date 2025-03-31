; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 569437

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child486 - child
    bread86 - bread-portion
    content18 - content-portion
    tray286 tray425 tray90 - tray
    table189 table426 table144 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray286 kitchen)
     (at tray425 kitchen)
     (at tray90 kitchen)
     (at_kitchen_bread bread86)
     (at_kitchen_content content18)
     (not_allergic_gluten child486)
     (waiting child486 table144)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child486)
    )
  )
)
