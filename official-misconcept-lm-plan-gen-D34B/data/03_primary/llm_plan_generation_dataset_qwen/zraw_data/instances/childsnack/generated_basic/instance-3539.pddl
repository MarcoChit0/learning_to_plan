; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 102312

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child104 child167 - child
    bread336 bread168 - bread-portion
    content204 content399 - content-portion
    tray424 tray270 - tray
    table348 table437 table144 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray424 kitchen)
     (at tray270 kitchen)
     (at_kitchen_bread bread336)
     (at_kitchen_bread bread168)
     (at_kitchen_content content204)
     (at_kitchen_content content399)
     (not_allergic_gluten child104)
     (not_allergic_gluten child167)
     (waiting child104 table144)
     (waiting child167 table348)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child104)
     (served child167)
    )
  )
)
