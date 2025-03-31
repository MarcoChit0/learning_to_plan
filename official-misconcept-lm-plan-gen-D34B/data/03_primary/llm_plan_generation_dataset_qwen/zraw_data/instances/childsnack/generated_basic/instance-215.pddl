; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 888109

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child259 child266 - child
    bread470 bread450 - bread-portion
    content84 content202 - content-portion
    tray345 tray391 tray19 - tray
    table325 table357 table313 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray345 kitchen)
     (at tray391 kitchen)
     (at tray19 kitchen)
     (at_kitchen_bread bread470)
     (at_kitchen_bread bread450)
     (at_kitchen_content content84)
     (at_kitchen_content content202)
     (not_allergic_gluten child259)
     (not_allergic_gluten child266)
     (waiting child259 table325)
     (waiting child266 table313)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child259)
     (served child266)
    )
  )
)
