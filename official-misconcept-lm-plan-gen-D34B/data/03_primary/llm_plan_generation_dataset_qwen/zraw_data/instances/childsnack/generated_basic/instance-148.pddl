; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 566646

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child440 child302 - child
    bread483 bread285 - bread-portion
    content390 content387 - content-portion
    tray377 tray386 tray152 tray391 - tray
    table30 table413 table70 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray377 kitchen)
     (at tray386 kitchen)
     (at tray152 kitchen)
     (at tray391 kitchen)
     (at_kitchen_bread bread483)
     (at_kitchen_bread bread285)
     (at_kitchen_content content390)
     (at_kitchen_content content387)
     (not_allergic_gluten child302)
     (not_allergic_gluten child440)
     (waiting child440 table70)
     (waiting child302 table70)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child440)
     (served child302)
    )
  )
)
