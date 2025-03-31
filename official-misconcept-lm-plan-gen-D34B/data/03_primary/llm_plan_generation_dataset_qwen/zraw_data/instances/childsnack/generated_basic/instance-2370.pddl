; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 655306

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child225 child140 - child
    bread404 bread276 - bread-portion
    content267 content277 - content-portion
    tray92 - tray
    table442 table84 table230 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray92 kitchen)
     (at_kitchen_bread bread404)
     (at_kitchen_bread bread276)
     (at_kitchen_content content267)
     (at_kitchen_content content277)
     (not_allergic_gluten child225)
     (not_allergic_gluten child140)
     (waiting child225 table442)
     (waiting child140 table84)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child225)
     (served child140)
    )
  )
)
