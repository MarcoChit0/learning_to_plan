; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 478209

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child397 - child
    bread246 - bread-portion
    content394 - content-portion
    tray83 tray481 tray245 tray55 - tray
    table135 table281 table377 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray83 kitchen)
     (at tray481 kitchen)
     (at tray245 kitchen)
     (at tray55 kitchen)
     (at_kitchen_bread bread246)
     (at_kitchen_content content394)
     (not_allergic_gluten child397)
     (waiting child397 table281)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child397)
    )
  )
)
