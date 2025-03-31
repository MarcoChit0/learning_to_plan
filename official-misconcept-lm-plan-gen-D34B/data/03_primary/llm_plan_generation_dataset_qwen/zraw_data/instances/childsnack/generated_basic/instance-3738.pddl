; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 676453

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child162 - child
    bread290 - bread-portion
    content185 - content-portion
    tray383 tray221 tray109 - tray
    table333 table122 table90 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray383 kitchen)
     (at tray221 kitchen)
     (at tray109 kitchen)
     (at_kitchen_bread bread290)
     (at_kitchen_content content185)
     (not_allergic_gluten child162)
     (waiting child162 table333)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child162)
    )
  )
)
