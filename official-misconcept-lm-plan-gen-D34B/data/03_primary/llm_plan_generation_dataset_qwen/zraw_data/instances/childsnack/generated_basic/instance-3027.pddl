; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 567440

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child434 child429 - child
    bread129 bread217 - bread-portion
    content58 content246 - content-portion
    tray473 tray360 tray340 tray446 - tray
    table404 table174 table468 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray473 kitchen)
     (at tray360 kitchen)
     (at tray340 kitchen)
     (at tray446 kitchen)
     (at_kitchen_bread bread129)
     (at_kitchen_bread bread217)
     (at_kitchen_content content58)
     (at_kitchen_content content246)
     (not_allergic_gluten child429)
     (not_allergic_gluten child434)
     (waiting child434 table174)
     (waiting child429 table404)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child434)
     (served child429)
    )
  )
)
