; child-snack task with 2 children and 0.65 gluten factor 
; constant factor of 1.3
; random seed: 596919

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child62 child9 - child
    bread276 bread62 - bread-portion
    content268 content318 - content-portion
    tray454 tray394 tray109 tray154 - tray
    table387 table274 table170 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray454 kitchen)
     (at tray394 kitchen)
     (at tray109 kitchen)
     (at tray154 kitchen)
     (at_kitchen_bread bread276)
     (at_kitchen_bread bread62)
     (at_kitchen_content content268)
     (at_kitchen_content content318)
     (no_gluten_bread bread276)
     (no_gluten_content content318)
     (allergic_gluten child62)
     (not_allergic_gluten child9)
     (waiting child62 table170)
     (waiting child9 table274)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child62)
     (served child9)
    )
  )
)
