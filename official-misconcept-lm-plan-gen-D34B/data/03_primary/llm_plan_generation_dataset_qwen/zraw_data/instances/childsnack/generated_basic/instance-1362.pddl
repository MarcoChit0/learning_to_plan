; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 121512

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child218 child108 - child
    bread445 bread264 - bread-portion
    content274 content115 - content-portion
    tray289 tray265 tray150 - tray
    table322 table318 table106 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray289 kitchen)
     (at tray265 kitchen)
     (at tray150 kitchen)
     (at_kitchen_bread bread445)
     (at_kitchen_bread bread264)
     (at_kitchen_content content274)
     (at_kitchen_content content115)
     (not_allergic_gluten child108)
     (not_allergic_gluten child218)
     (waiting child218 table322)
     (waiting child108 table318)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child218)
     (served child108)
    )
  )
)
