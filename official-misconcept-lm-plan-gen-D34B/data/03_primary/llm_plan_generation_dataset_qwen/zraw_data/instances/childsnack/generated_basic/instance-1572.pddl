; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 521844

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child246 child455 - child
    bread349 bread239 - bread-portion
    content435 content310 - content-portion
    tray166 tray418 tray203 tray263 - tray
    table84 table451 table196 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray166 kitchen)
     (at tray418 kitchen)
     (at tray203 kitchen)
     (at tray263 kitchen)
     (at_kitchen_bread bread349)
     (at_kitchen_bread bread239)
     (at_kitchen_content content435)
     (at_kitchen_content content310)
     (not_allergic_gluten child455)
     (not_allergic_gluten child246)
     (waiting child246 table84)
     (waiting child455 table196)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child246)
     (served child455)
    )
  )
)
