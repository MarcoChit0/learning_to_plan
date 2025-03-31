; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 376309

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child45 child105 - child
    bread483 bread332 - bread-portion
    content48 content222 - content-portion
    tray323 tray283 - tray
    table36 table459 table436 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray323 kitchen)
     (at tray283 kitchen)
     (at_kitchen_bread bread483)
     (at_kitchen_bread bread332)
     (at_kitchen_content content48)
     (at_kitchen_content content222)
     (no_gluten_bread bread332)
     (no_gluten_content content48)
     (allergic_gluten child45)
     (not_allergic_gluten child105)
     (waiting child45 table436)
     (waiting child105 table459)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child45)
     (served child105)
    )
  )
)
