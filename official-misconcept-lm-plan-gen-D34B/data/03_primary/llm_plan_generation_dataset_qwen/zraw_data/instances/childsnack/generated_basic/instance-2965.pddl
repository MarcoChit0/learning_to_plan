; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 523095

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child210 - child
    bread188 - bread-portion
    content215 - content-portion
    tray88 tray291 tray306 tray240 - tray
    table115 table322 table469 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray88 kitchen)
     (at tray291 kitchen)
     (at tray306 kitchen)
     (at tray240 kitchen)
     (at_kitchen_bread bread188)
     (at_kitchen_content content215)
     (not_allergic_gluten child210)
     (waiting child210 table115)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child210)
    )
  )
)
