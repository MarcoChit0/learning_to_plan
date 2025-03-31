; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 457276

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child50 - child
    bread28 - bread-portion
    content418 - content-portion
    tray356 tray278 tray243 - tray
    table299 table454 table221 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray356 kitchen)
     (at tray278 kitchen)
     (at tray243 kitchen)
     (at_kitchen_bread bread28)
     (at_kitchen_content content418)
     (not_allergic_gluten child50)
     (waiting child50 table221)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child50)
    )
  )
)
