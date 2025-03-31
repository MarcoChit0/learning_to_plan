; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 877621

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child134 child276 child192 - child
    bread222 bread7 bread310 - bread-portion
    content363 content443 content59 - content-portion
    tray286 - tray
    table393 table324 table377 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray286 kitchen)
     (at_kitchen_bread bread222)
     (at_kitchen_bread bread7)
     (at_kitchen_bread bread310)
     (at_kitchen_content content363)
     (at_kitchen_content content443)
     (at_kitchen_content content59)
     (not_allergic_gluten child192)
     (not_allergic_gluten child134)
     (not_allergic_gluten child276)
     (waiting child134 table324)
     (waiting child276 table377)
     (waiting child192 table377)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child134)
     (served child276)
     (served child192)
    )
  )
)
