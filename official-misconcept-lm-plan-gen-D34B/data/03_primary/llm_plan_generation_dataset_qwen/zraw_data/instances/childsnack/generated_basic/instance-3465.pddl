; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 349215

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child258 child104 - child
    bread143 bread101 - bread-portion
    content301 content107 - content-portion
    tray468 tray268 - tray
    table420 table375 table61 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray468 kitchen)
     (at tray268 kitchen)
     (at_kitchen_bread bread143)
     (at_kitchen_bread bread101)
     (at_kitchen_content content301)
     (at_kitchen_content content107)
     (no_gluten_bread bread101)
     (no_gluten_content content107)
     (allergic_gluten child104)
     (not_allergic_gluten child258)
     (waiting child258 table375)
     (waiting child104 table420)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child258)
     (served child104)
    )
  )
)
