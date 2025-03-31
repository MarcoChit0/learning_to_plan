; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 699245

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child433 child103 - child
    bread34 bread301 - bread-portion
    content255 content323 - content-portion
    tray491 tray11 - tray
    table84 table281 table195 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray491 kitchen)
     (at tray11 kitchen)
     (at_kitchen_bread bread34)
     (at_kitchen_bread bread301)
     (at_kitchen_content content255)
     (at_kitchen_content content323)
     (no_gluten_bread bread301)
     (no_gluten_content content255)
     (allergic_gluten child103)
     (not_allergic_gluten child433)
     (waiting child433 table84)
     (waiting child103 table195)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child433)
     (served child103)
    )
  )
)
