; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 845923

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child50 child291 - child
    bread388 bread444 - bread-portion
    content360 content366 - content-portion
    tray321 tray378 tray492 tray109 - tray
    table221 table387 table92 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray321 kitchen)
     (at tray378 kitchen)
     (at tray492 kitchen)
     (at tray109 kitchen)
     (at_kitchen_bread bread388)
     (at_kitchen_bread bread444)
     (at_kitchen_content content360)
     (at_kitchen_content content366)
     (not_allergic_gluten child50)
     (not_allergic_gluten child291)
     (waiting child50 table92)
     (waiting child291 table387)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child50)
     (served child291)
    )
  )
)
