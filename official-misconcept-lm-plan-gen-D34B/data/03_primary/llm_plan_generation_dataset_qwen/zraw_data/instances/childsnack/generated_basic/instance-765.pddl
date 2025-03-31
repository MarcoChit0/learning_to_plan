; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 168607

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child196 child212 - child
    bread71 bread122 - bread-portion
    content146 content204 - content-portion
    tray350 tray93 - tray
    table37 table79 table341 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray350 kitchen)
     (at tray93 kitchen)
     (at_kitchen_bread bread71)
     (at_kitchen_bread bread122)
     (at_kitchen_content content146)
     (at_kitchen_content content204)
     (no_gluten_bread bread122)
     (no_gluten_content content204)
     (allergic_gluten child212)
     (not_allergic_gluten child196)
     (waiting child196 table37)
     (waiting child212 table37)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child196)
     (served child212)
    )
  )
)
