; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 523162

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child214 child303 - child
    bread74 bread304 - bread-portion
    content395 content237 - content-portion
    tray78 tray380 tray140 - tray
    table492 table470 table490 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray78 kitchen)
     (at tray380 kitchen)
     (at tray140 kitchen)
     (at_kitchen_bread bread74)
     (at_kitchen_bread bread304)
     (at_kitchen_content content395)
     (at_kitchen_content content237)
     (not_allergic_gluten child214)
     (not_allergic_gluten child303)
     (waiting child214 table470)
     (waiting child303 table492)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child214)
     (served child303)
    )
  )
)
