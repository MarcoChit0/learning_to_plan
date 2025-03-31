; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 621003

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child251 child346 - child
    bread440 bread473 - bread-portion
    content263 content337 - content-portion
    tray481 tray375 tray448 - tray
    table314 table412 table397 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray481 kitchen)
     (at tray375 kitchen)
     (at tray448 kitchen)
     (at_kitchen_bread bread440)
     (at_kitchen_bread bread473)
     (at_kitchen_content content263)
     (at_kitchen_content content337)
     (no_gluten_bread bread473)
     (no_gluten_content content337)
     (allergic_gluten child251)
     (not_allergic_gluten child346)
     (waiting child251 table412)
     (waiting child346 table314)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child251)
     (served child346)
    )
  )
)
