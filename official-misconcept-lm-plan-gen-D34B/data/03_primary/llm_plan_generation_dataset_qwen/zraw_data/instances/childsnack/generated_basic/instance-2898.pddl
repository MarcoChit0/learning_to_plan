; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 108745

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child347 child145 - child
    bread192 bread184 - bread-portion
    content276 content476 - content-portion
    tray158 tray68 tray104 - tray
    table492 table447 table113 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray158 kitchen)
     (at tray68 kitchen)
     (at tray104 kitchen)
     (at_kitchen_bread bread192)
     (at_kitchen_bread bread184)
     (at_kitchen_content content276)
     (at_kitchen_content content476)
     (no_gluten_bread bread184)
     (no_gluten_content content476)
     (allergic_gluten child145)
     (not_allergic_gluten child347)
     (waiting child347 table113)
     (waiting child145 table492)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child347)
     (served child145)
    )
  )
)
