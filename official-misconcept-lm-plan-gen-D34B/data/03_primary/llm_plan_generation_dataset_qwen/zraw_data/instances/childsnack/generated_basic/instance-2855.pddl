; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 845647

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child167 - child
    bread398 - bread-portion
    content154 - content-portion
    tray233 tray425 - tray
    table89 table261 table119 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray233 kitchen)
     (at tray425 kitchen)
     (at_kitchen_bread bread398)
     (at_kitchen_content content154)
     (not_allergic_gluten child167)
     (waiting child167 table119)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child167)
    )
  )
)
