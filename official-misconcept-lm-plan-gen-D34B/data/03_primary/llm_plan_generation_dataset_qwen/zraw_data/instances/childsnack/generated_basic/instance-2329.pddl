; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 655212

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child86 - child
    bread412 - bread-portion
    content403 - content-portion
    tray189 - tray
    table122 table176 table23 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray189 kitchen)
     (at_kitchen_bread bread412)
     (at_kitchen_content content403)
     (not_allergic_gluten child86)
     (waiting child86 table176)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child86)
    )
  )
)
