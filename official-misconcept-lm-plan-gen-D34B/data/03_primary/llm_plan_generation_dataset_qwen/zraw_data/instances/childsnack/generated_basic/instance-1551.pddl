; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 221339

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child338 child479 - child
    bread313 bread299 - bread-portion
    content86 content42 - content-portion
    tray362 tray412 tray475 tray253 - tray
    table82 table128 table0 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray362 kitchen)
     (at tray412 kitchen)
     (at tray475 kitchen)
     (at tray253 kitchen)
     (at_kitchen_bread bread313)
     (at_kitchen_bread bread299)
     (at_kitchen_content content86)
     (at_kitchen_content content42)
     (not_allergic_gluten child479)
     (not_allergic_gluten child338)
     (waiting child338 table128)
     (waiting child479 table0)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child338)
     (served child479)
    )
  )
)
