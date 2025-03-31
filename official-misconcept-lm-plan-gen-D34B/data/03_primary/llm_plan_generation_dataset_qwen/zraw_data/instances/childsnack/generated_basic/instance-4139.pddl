; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 188738

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child425 child144 - child
    bread485 bread342 - bread-portion
    content205 content322 - content-portion
    tray289 tray94 tray236 - tray
    table24 table457 table317 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray289 kitchen)
     (at tray94 kitchen)
     (at tray236 kitchen)
     (at_kitchen_bread bread485)
     (at_kitchen_bread bread342)
     (at_kitchen_content content205)
     (at_kitchen_content content322)
     (not_allergic_gluten child144)
     (not_allergic_gluten child425)
     (waiting child425 table317)
     (waiting child144 table24)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child425)
     (served child144)
    )
  )
)
