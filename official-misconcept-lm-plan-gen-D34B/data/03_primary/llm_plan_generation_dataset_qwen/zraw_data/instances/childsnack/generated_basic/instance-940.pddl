; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 156456

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child406 child84 child435 - child
    bread423 bread391 bread463 - bread-portion
    content207 content343 content439 - content-portion
    tray32 tray305 tray296 tray205 - tray
    table86 table421 table77 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray32 kitchen)
     (at tray305 kitchen)
     (at tray296 kitchen)
     (at tray205 kitchen)
     (at_kitchen_bread bread423)
     (at_kitchen_bread bread391)
     (at_kitchen_bread bread463)
     (at_kitchen_content content207)
     (at_kitchen_content content343)
     (at_kitchen_content content439)
     (not_allergic_gluten child435)
     (not_allergic_gluten child406)
     (not_allergic_gluten child84)
     (waiting child406 table86)
     (waiting child84 table421)
     (waiting child435 table421)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child406)
     (served child84)
     (served child435)
    )
  )
)
