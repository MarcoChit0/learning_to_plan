; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 10377

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child163 child478 child41 - child
    bread144 bread498 bread496 - bread-portion
    content249 content191 content465 - content-portion
    tray280 - tray
    table226 table359 table241 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray280 kitchen)
     (at_kitchen_bread bread144)
     (at_kitchen_bread bread498)
     (at_kitchen_bread bread496)
     (at_kitchen_content content249)
     (at_kitchen_content content191)
     (at_kitchen_content content465)
     (not_allergic_gluten child163)
     (not_allergic_gluten child41)
     (not_allergic_gluten child478)
     (waiting child163 table226)
     (waiting child478 table226)
     (waiting child41 table241)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child163)
     (served child478)
     (served child41)
    )
  )
)
