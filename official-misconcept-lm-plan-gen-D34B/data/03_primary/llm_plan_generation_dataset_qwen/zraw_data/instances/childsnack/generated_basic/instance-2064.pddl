; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 191261

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child349 - child
    bread459 - bread-portion
    content368 - content-portion
    tray450 tray405 tray0 tray121 - tray
    table89 table419 table170 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray450 kitchen)
     (at tray405 kitchen)
     (at tray0 kitchen)
     (at tray121 kitchen)
     (at_kitchen_bread bread459)
     (at_kitchen_content content368)
     (not_allergic_gluten child349)
     (waiting child349 table89)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child349)
    )
  )
)
