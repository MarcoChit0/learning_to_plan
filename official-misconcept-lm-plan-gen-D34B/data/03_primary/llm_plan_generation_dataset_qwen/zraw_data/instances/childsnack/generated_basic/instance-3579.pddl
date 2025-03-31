; child-snack task with 3 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 468273

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child144 child223 child166 - child
    bread159 bread121 bread87 - bread-portion
    content132 content257 content152 - content-portion
    tray181 tray203 tray244 - tray
    table323 table332 table439 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray181 kitchen)
     (at tray203 kitchen)
     (at tray244 kitchen)
     (at_kitchen_bread bread159)
     (at_kitchen_bread bread121)
     (at_kitchen_bread bread87)
     (at_kitchen_content content132)
     (at_kitchen_content content257)
     (at_kitchen_content content152)
     (no_gluten_bread bread121)
     (no_gluten_content content257)
     (allergic_gluten child166)
     (not_allergic_gluten child144)
     (not_allergic_gluten child223)
     (waiting child144 table439)
     (waiting child223 table439)
     (waiting child166 table323)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child144)
     (served child223)
     (served child166)
    )
  )
)
