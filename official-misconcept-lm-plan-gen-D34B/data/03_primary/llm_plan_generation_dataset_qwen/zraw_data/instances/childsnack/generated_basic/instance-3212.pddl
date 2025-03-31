; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 977167

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child363 child411 - child
    bread339 bread159 - bread-portion
    content441 content327 - content-portion
    tray11 tray158 tray248 tray481 - tray
    table125 table164 table476 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray11 kitchen)
     (at tray158 kitchen)
     (at tray248 kitchen)
     (at tray481 kitchen)
     (at_kitchen_bread bread339)
     (at_kitchen_bread bread159)
     (at_kitchen_content content441)
     (at_kitchen_content content327)
     (no_gluten_bread bread339)
     (no_gluten_content content327)
     (allergic_gluten child411)
     (not_allergic_gluten child363)
     (waiting child363 table476)
     (waiting child411 table125)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child363)
     (served child411)
    )
  )
)
