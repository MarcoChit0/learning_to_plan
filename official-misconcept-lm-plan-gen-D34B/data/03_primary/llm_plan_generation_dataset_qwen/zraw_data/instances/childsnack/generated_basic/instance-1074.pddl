; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 339626

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child170 child397 - child
    bread121 bread105 - bread-portion
    content54 content442 - content-portion
    tray132 tray393 tray343 - tray
    table70 table481 table170 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray132 kitchen)
     (at tray393 kitchen)
     (at tray343 kitchen)
     (at_kitchen_bread bread121)
     (at_kitchen_bread bread105)
     (at_kitchen_content content54)
     (at_kitchen_content content442)
     (not_allergic_gluten child397)
     (not_allergic_gluten child170)
     (waiting child170 table70)
     (waiting child397 table70)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child170)
     (served child397)
    )
  )
)
