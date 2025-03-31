; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 403837

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child323 child80 - child
    bread312 bread444 - bread-portion
    content232 content401 - content-portion
    tray47 tray293 - tray
    table246 table212 table180 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray47 kitchen)
     (at tray293 kitchen)
     (at_kitchen_bread bread312)
     (at_kitchen_bread bread444)
     (at_kitchen_content content232)
     (at_kitchen_content content401)
     (not_allergic_gluten child80)
     (not_allergic_gluten child323)
     (waiting child323 table246)
     (waiting child80 table180)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child323)
     (served child80)
    )
  )
)
