; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 188203

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child116 - child
    bread387 - bread-portion
    content164 - content-portion
    tray245 - tray
    table282 table416 table361 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray245 kitchen)
     (at_kitchen_bread bread387)
     (at_kitchen_content content164)
     (not_allergic_gluten child116)
     (waiting child116 table416)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child116)
    )
  )
)
